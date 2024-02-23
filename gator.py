from torch.distributions.normal import Normal
from torch.autograd import Variable
import torch
import torch.optim as optim

def reparametrization_trick_gaussian(mu, sigma2, device, n_samples):
    # Expand the mean tensor to match the required shape
    mu_expanded = mu.expand(n_samples, -1, -1, -1)
    # Create a tensor for sigma2
    sigma2_tensor = torch.tensor(sigma2).to(device)
    # Create a Normal distribution with the given mean and standard deviation
    normal_dist = Normal(mu_expanded, torch.sqrt(sigma2_tensor))
    # Sample from the Normal distribution
    samples = normal_dist.sample()
    return samples


def validity_loss(x0,x_new,model):
    output_x0 = model(x0)
    output_x_new = model(x_new)
    # Compute 1 - model(x_new)
    one_minus_output_x_new = 1 - output_x_new
    # Compute the difference between model(x0) and 1 - model(x_new)
    difference = output_x0 - one_minus_output_x_new
    # Calculate the norm of the difference tensor
    # You can choose different norms such as L1, L2, etc.
    # Here, we compute the L2 norm using torch.norm
    norm_difference = torch.norm(difference, p=2)
    return norm_difference

def croco(model,autoencoder,delta,x,weights,n_samples,lr,sigma2,robustness_target,robustness_epsilon,n_iter,t,m):
    device = "cpu"
    # Input example as a tensor 
    x0 = torch.from_numpy(x).float().to(device)
    # Tensor init perturb
    delta = torch.from_numpy(delta)

    # Target classes are 1, one hot encoded -> [0,1]
    y_target_class = torch.tensor([0,1]).float().to(device)
    y_target = y_target_class[1]
    G_target = torch.tensor(y_target).float().to(device)
    # Init weights value
    rob_w = torch.tensor(weights[0]).float()
    val_w = torch.tensor(weights[1]).float()
    prox_w = torch.tensor(weights[2]).float()
    # Init perturb 
    Perturb = Variable(torch.clone(delta.to(device)), requires_grad=True)
    x_cf_new = (x0+Perturb).to(device)
    # Set optimizer 
    optimizer = optim.Adam([Perturb], lr, amsgrad=True)
    # MSE loss for class term 
    loss_fn = torch.nn.MSELoss()

    #get samples
    random_samples = reparametrization_trick_gaussian(autoencoder.encode(x_cf_new.float()), sigma2, device, n_samples=n_samples)
    random_samples = autoencoder.decode(random_samples)
    G = random_samples

    # Compute robustness constraint term 
    #compute_robutness = (m + torch.mean(G_target- model((G_new).float())[:,1-pred_class])) / (1-t)
    compute_robutness = (m + torch.mean(G_target - model((G).float())[1-pred_class])) / (1-t)

    #Lambda = []
    #Dist = []
    #Rob = []
    hist={}
    hist['robustness'],hist["validity"],hist["proximity"],hist["loss"] = [],[],[],[]
    #hist["perturbations"]
    while (f_x <=t) and (compute_robutness > robustness_target + robustness_epsilon) : 
        it=0
        for it in range(n_iter) :
            optimizer.zero_grad()
            x_cf_new = x0+Perturb
            # Take random samples 
            random_samples = reparametrization_trick_gaussian(autoencoder.encode(x_cf_new.float()), sigma2, device, n_samples=n_samples)
            random_samples = autoencoder.decode(random_samples)
            #invalidation_rate = compute_invalidation_rate(model, random_samples)
            # New perturbated group translated 
            G = random_samples
            # Compute (m + theta) / (1-t)
            mean_proba =  torch.mean(model((G).float())[:,pred_class])
            compute_robutness = (m + mean_proba) /(1-t)
            # Diff between robustness and targer robustness 
            robustness_invalidation = compute_robutness - robustness_target            
            # Overall loss function 
            #loss = rob_w*robustness_invalidation**2 + val_w*loss_fn(f_x_binary,y_target_class) + prox_w* torch.norm(Perturb,p=1)
            loss = rob_w*robustness_invalidation**2 + val_w*validity_loss(x0,x_cf_new.float(),model) + prox_w* torch.norm(Perturb,p=1)
            loss.backward()
            optimizer.step()
            
            hist["robustness"].append((robustness_invalidation**2).item())
            hist["validity"].append(loss_fn(f_x_binary,y_target_class).item())
            hist["proximity"].append( torch.norm(Perturb,p=1).item())
            hist["loss"].append(loss.item())

            it += 1
        if (f_x > t) and ((compute_robutness < robustness_target + robustness_epsilon))  :
            print("Counterfactual Explanation Found")
            break   
        
        # this behaviour is not in the paper and is concerning
        #lamb -= 0.25

        # Stop if no solution found for different lambda values 
        #if lamb <=0 :
        #    print("No Counterfactual Explanation Found for these lambda values")
        #    break

    final_perturb = Perturb.clone()
    x_new =(x0 + final_perturb).float().detach()
    return x_new,hist
            
            
 